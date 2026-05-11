
import numpy as np
import matplotlib.pyplot as plt

def save_joint_video_action_attention_matrix(
    T=8,
    T_V=4,
    save_path="joint_video_action_attention.png"
):
    r = T / T_V
    N = T_V + T  # video + action (time-level)

    # ----------------------------------
    # build joint mask [N, N]
    # order: [v0 ... v_{T_V-1}, a0 ... a_{T-1}]
    # ----------------------------------
    mat = np.zeros((N, N), dtype=np.float32)

    # Video -> Video (causal)
    for tau_q in range(T_V):
        mat[tau_q, :tau_q + 1] = 1.0

    # Action -> Action (causal)
    for t_q in range(T):
        mat[T_V + t_q, T_V : T_V + t_q + 1] = 1.0

    # Action -> Video (future video)
    for t_q in range(T):
        tau_min = int(t_q * T_V / T)
        mat[T_V + t_q, tau_min : T_V] = 1.0

    # Video -> Action (current + future action)
    for tau_q in range(T_V):
        t_min = int(tau_q * T / T_V)
        mat[tau_q, T_V + t_min : T_V + T] = 1.0

    # ----------------------------------
    # plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(
        mat,
        cmap="gray_r",
        interpolation="nearest"
    )

    # ticks
    x_labels = [f"v{τ}" for τ in range(T_V)] + [f"a{t}" for t in range(T)]
    y_labels = x_labels

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels)

    ax.set_title("Joint Video–Action Attention Mask")
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")

    # block separator lines
    ax.axhline(T_V - 0.5, color="red", linewidth=1)
    ax.axvline(T_V - 0.5, color="red", linewidth=1)

    # grid
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved to {save_path}")



save_joint_video_action_attention_matrix(
    T=16,
    T_V=4,
    save_path="video_action_attention.png"
)