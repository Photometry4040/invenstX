"""
Transformer 기반 주가 예측 모델
Multi-head Attention을 활용한 시계열 패턴 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import math
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Transformer 모델 설정"""
    input_dim: int = 23          # 입력 특징 수
    d_model: int = 512           # 모델 차원
    nhead: int = 8               # Attention head 수
    num_layers: int = 6          # Encoder layer 수
    dim_feedforward: int = 2048  # Feedforward 네트워크 차원
    dropout: float = 0.1         # Dropout 비율
    max_seq_length: int = 60     # 최대 시퀀스 길이
    num_classes: int = 3         # 출력 클래스 수 (Buy, Hold, Sell)
    learning_rate: float = 0.0001
    warmup_steps: int = 4000


class PositionalEncoding(nn.Module):
    """위치 인코딩 레이어"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        # 위치 인코딩 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention 메커니즘"""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query, key, value: [seq_len, batch_size, d_model]
            mask: [seq_len, seq_len]
        """
        batch_size = query.size(1)

        # Linear transformations and split into heads
        Q = self.W_q(query).view(-1, batch_size, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(-1, batch_size, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(-1, batch_size, self.nhead, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            -1, batch_size, self.d_model
        )

        # Final linear transformation
        output = self.W_o(context)

        return output


class StockPriceTransformer(nn.Module):
    """
    주가 예측을 위한 Transformer 모델
    시계열 패턴을 학습하여 Buy/Hold/Sell 신호 생성
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # 입력 프로젝션
        self.input_projection = nn.Linear(config.input_dim, config.d_model)

        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=False  # [seq_len, batch, features]
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model)
        )

        # 시간적 특징 추출을 위한 추가 레이어
        self.temporal_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            padding=1
        )

        # 출력 레이어
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, config.num_classes)
        )

        # 신뢰도 예측 레이어
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 가중치 초기화
        self._init_weights()

        # 학습 가능한 파라미터 수 계산
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {self.num_parameters:,}")

    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """패딩 마스크 생성"""
        mask = torch.ones(seq_len, seq_len, device=device)
        return mask

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            src: [batch_size, seq_len, input_dim]
            src_mask: Optional mask tensor

        Returns:
            predictions: [batch_size, num_classes]
            confidence: [batch_size, 1]
        """
        # 차원 변환: [batch, seq, features] -> [seq, batch, features]
        src = src.transpose(0, 1)

        # 입력 프로젝션
        src = self.input_projection(src)

        # 위치 인코딩
        src = self.pos_encoder(src)

        # Transformer Encoder
        if src_mask is None:
            src_mask = self.create_padding_mask(
                src.size(0), src.size(1), src.device
            )

        output = self.transformer_encoder(src, src_mask)

        # Temporal convolution (선택적)
        # [seq, batch, features] -> [batch, features, seq]
        conv_input = output.transpose(0, 1).transpose(1, 2)
        conv_output = self.temporal_conv(conv_input)
        # [batch, features, seq] -> [seq, batch, features]
        output = conv_output.transpose(1, 2).transpose(0, 1)

        # 마지막 시간 스텝의 표현 사용
        final_output = output[-1, :, :]

        # 예측 및 신뢰도
        predictions = self.decoder(final_output)
        confidence = self.confidence_head(final_output)

        return predictions, confidence

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[int, float]:
        """
        예측과 신뢰도 반환

        Returns:
            action: 0 (Buy), 1 (Hold), 2 (Sell)
            confidence: 0-1 사이의 신뢰도
        """
        self.eval()
        with torch.no_grad():
            predictions, confidence = self.forward(x)
            probabilities = F.softmax(predictions, dim=-1)
            action = torch.argmax(probabilities, dim=-1)

        return action.item(), confidence.item()


class TransformerTrainer:
    """Transformer 모델 학습기"""

    def __init__(self,
                 model: StockPriceTransformer,
                 config: TransformerConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=100000
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 학습 히스토리
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        """Warmup을 포함한 선형 스케줄러"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) /
                      float(max(1, num_training_steps - num_warmup_steps)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_epoch(self, dataloader, epoch: int):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions, confidence = self.model(data)

            # Loss calculation
            loss = self.criterion(predictions, targets)

            # Confidence loss (optional)
            # 높은 신뢰도일 때 잘못된 예측에 대한 페널티
            pred_labels = torch.argmax(predictions, dim=-1)
            is_correct = (pred_labels == targets).float()
            confidence_loss = F.binary_cross_entropy(
                confidence.squeeze(),
                is_correct,
                reduction='mean'
            )
            loss = loss + 0.1 * confidence_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Statistics
            total_loss += loss.item()
            total_correct += (pred_labels == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                          f'Accuracy: {total_correct/total_samples:.4f}')

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        self.train_losses.append(avg_loss)

        return avg_loss, accuracy

    def validate(self, dataloader):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                predictions, confidence = self.model(data)
                loss = self.criterion(predictions, targets)

                pred_labels = torch.argmax(predictions, dim=-1)
                total_loss += loss.item()
                total_correct += (pred_labels == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        self.val_losses.append(avg_loss)

        # Best model 저장
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best_transformer_model.pth')

        return avg_loss, accuracy

    def save_checkpoint(self, filepath: str):
        """모델 체크포인트 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """모델 체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        logger.info(f"Model loaded from {filepath}")


class TransformerPredictor:
    """Transformer 모델을 사용한 예측기"""

    def __init__(self, model_path: str, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else
                                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        # 모델 로드
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']

        self.model = StockPriceTransformer(config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded from {model_path}")

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        예측 수행

        Args:
            features: [seq_len, num_features] numpy array

        Returns:
            action: 0 (Buy), 1 (Hold), 2 (Sell)
            confidence: 0-1 사이의 신뢰도
        """
        # Tensor 변환
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # 예측
        with torch.no_grad():
            predictions, confidence = self.model(features_tensor)
            probabilities = F.softmax(predictions, dim=-1)
            action = torch.argmax(probabilities, dim=-1)

        return action.item(), confidence.item()

    def predict_batch(self, features_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 예측

        Args:
            features_batch: [batch_size, seq_len, num_features]

        Returns:
            actions: [batch_size]
            confidences: [batch_size]
        """
        features_tensor = torch.FloatTensor(features_batch).to(self.device)

        with torch.no_grad():
            predictions, confidences = self.model(features_tensor)
            probabilities = F.softmax(predictions, dim=-1)
            actions = torch.argmax(probabilities, dim=-1)

        return actions.cpu().numpy(), confidences.cpu().numpy()


def create_sample_data(batch_size: int = 32,
                      seq_len: int = 60,
                      num_features: int = 23) -> Tuple[torch.Tensor, torch.Tensor]:
    """샘플 데이터 생성 (테스트용)"""
    # 랜덤 특징
    data = torch.randn(batch_size, seq_len, num_features)

    # 랜덤 레이블 (0: Buy, 1: Hold, 2: Sell)
    labels = torch.randint(0, 3, (batch_size,))

    return data, labels


def main():
    """테스트 함수"""
    # 설정
    config = TransformerConfig(
        input_dim=23,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 생성
    model = StockPriceTransformer(config)
    print(f"Model created with {model.num_parameters:,} parameters")

    # 샘플 데이터
    data, labels = create_sample_data()
    data = data.to(device)
    labels = labels.to(device)

    # Forward pass
    model = model.to(device)
    predictions, confidence = model(data)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Confidence shape: {confidence.shape}")

    # 예측 결과
    actions = torch.argmax(predictions, dim=-1)
    print(f"\nSample predictions: {actions[:10]}")
    print(f"Sample confidences: {confidence[:10].squeeze()}")

    # 학습 예시
    trainer = TransformerTrainer(model, config, device)

    # 간단한 학습 루프
    for epoch in range(3):
        # 더미 데이터로더 시뮬레이션
        dummy_dataloader = [(create_sample_data()) for _ in range(10)]

        loss, acc = trainer.train_epoch(dummy_dataloader, epoch)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.4f}")


if __name__ == "__main__":
    main()