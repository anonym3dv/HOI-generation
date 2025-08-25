# options/train_options.py
import argparse

class TrainTexMotMatchOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # --- existing args ---
        self.parser.add_argument('--name', type=str, default="test")
        self.parser.add_argument('--gpu_id', type=int, default=-1)
        self.parser.add_argument("--dataset_name", type=str, required=True,
                                 choices=["arctic", "grab", "h2o"],
                                 help="Dataset name (loads ../configs/dataset/<name>.yaml)")
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--max_epoch', type=int, default=1000)
        self.parser.add_argument('--lr', type=float, default=1e-4)

        self.parser.add_argument('--log_every', type=int, default=50)
        self.parser.add_argument('--save_every_e', type=int, default=50)
        self.parser.add_argument('--eval_every_e', type=int, default=5)
        self.parser.add_argument('--save_latest', type=int, default=500)

        self.parser.add_argument('--latent_dim', type=int, default=512)
        self.parser.add_argument('--dim_text_hidden', type=int, default=512)
        self.parser.add_argument('--max_text_len', type=int, default=35)

        self.parser.add_argument('--dim_motion_latent', type=int, default=512,
                                 help='Transformer d_model + text latent')
        self.parser.add_argument('--num_heads', type=int, default=8)
        self.parser.add_argument('--num_layers', type=int, default=6)

       
        self.parser.add_argument('--dim_pose', type=int, required=True,
                                 help='Total motion feature dim (e.g., lhand+rhand+obj)')
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help='Transformer dropout for motion encoder')

        # Optional flags
        self.parser.add_argument('--is_continue', action="store_true")
        self.parser.add_argument('--negative_margin', type=float, default=3.0)
        self.parser.add_argument('--feat_bias', type=float, default=5.0)
        self.parser.add_argument('--unit_length', type=int, default=4)

    def parse(self):
        opt = self.parser.parse_args()
        opt.is_train = True

        
        assert opt.dim_pose > 0, "--dim_pose must be > 0"
        assert opt.dim_motion_latent > 0, "--dim_motion_latent must be > 0"
        assert opt.num_heads > 0 and opt.num_layers > 0, "heads/layers must be > 0"
        assert 0.0 <= opt.dropout < 1.0, "--dropout must be in [0,1)"
        assert opt.dim_motion_latent % opt.num_heads == 0, \
            "--dim_motion_latent must be divisible by --num_heads"

        return opt
