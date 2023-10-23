from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')

    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_gamma', type=int, default=0.9)

    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--hidden_dim', type=int, default=16)
    p.add_argument('--act_fn', type=str, default='relu')

    return p.parse_args()
