import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. 그래프 데이터 구조 ────────────────────────────────────────────────────
# 노드 특징 행렬  X  : shape (N, F)  — N: 노드 수, F: 특징 수
# 인접 행렬      A  : shape (N, N)  — A[i][j]=1 이면 i↔j 간선 존재

class GraphData:
    def __init__(self, X: torch.Tensor, A: torch.Tensor, y: torch.Tensor = None):
        self.X = X          # 노드 특징
        self.A = A          # 인접 행렬
        self.y = y          # 레이블 (노드 분류용)


# ── 2. 정규화된 인접 행렬 계산 ───────────────────────────────────────────────
# GCN 논문(Kipf & Welling, 2017) 수식:
#   Â = D^{-1/2} (A + I) D^{-1/2}
def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    A_hat = A + torch.eye(A.size(0))            # 자기 자신 루프 추가
    D = A_hat.sum(dim=1)                        # 차수(degree) 벡터
    D_inv_sqrt = torch.diag(D.pow(-0.5))        # D^{-1/2}
    return D_inv_sqrt @ A_hat @ D_inv_sqrt      # Â


# ── 3. GCN 레이어 ────────────────────────────────────────────────────────────
# 메시지 전달 공식:
#   H^{(l+1)} = σ( Â · H^{(l)} · W^{(l)} )
class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # 이웃 정보 집계(Aggregation) → 변환(Transformation)
        return self.W(A_norm @ H)


# ── 4. 2-레이어 GCN 모델 ─────────────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden)
        self.layer2 = GCNLayer(hidden, num_classes)
        self.dropout = dropout

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1(X, A_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(h, A_norm)
        return F.log_softmax(h, dim=1)


# ── 5. 학습 루프 ─────────────────────────────────────────────────────────────
def train(model, graph, A_norm, optimizer, epochs=200):
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(graph.X, A_norm)
        loss = F.nll_loss(out, graph.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            pred = out.argmax(dim=1)
            acc  = (pred == graph.y).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")


# ── 6. 간단한 실험 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    N          = 6      # 노드 수
    F_in       = 4      # 입력 특징 차원
    num_classes = 3     # 클래스 수

    # 임의 그래프 생성
    X = torch.randn(N, F_in)
    A = torch.tensor([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ], dtype=torch.float)
    y = torch.tensor([0, 0, 1, 1, 2, 2])   # 노드 레이블

    graph  = GraphData(X, A, y)
    A_norm = normalize_adjacency(A)

    # 모델 / 옵티마이저
    model     = GCN(in_features=F_in, hidden=8, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("=== GCN 학습 시작 ===")
    train(model, graph, A_norm, optimizer, epochs=200)

    # 추론
    model.eval()
    with torch.no_grad():
        logits = model(graph.X, A_norm)
        preds  = logits.argmax(dim=1)
        print(f"\n예측 레이블: {preds.tolist()}")
        print(f"실제 레이블: {y.tolist()}")
