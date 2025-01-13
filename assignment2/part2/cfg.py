import argparse
import os

def get_config():
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, default="./assets/book_EN_grimms_fairy_tales.txt", help="Path to a .txt file to train on")
    parser.add_argument('--model_type', type=str, default='gpt-mini', help="Define the gpt2 version to be initialised")
    parser.add_argument('--block_size', type=int, default=128, help='Specify block size of input sequences')
    parser.add_argument('--use_pretrained', type=bool, default=False, help='Boolean whether to use pretrained huggingface weights.')
    parser.add_argument('--abs_emb', action="store_true", help='use absolute position embedding')
    # Training
    parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--generate_batch_size', type=int, default=5, help='Batch size for generated sentences in callback')
    parser.add_argument('--generate_every_n_steps', type=int, default=1000, help='Every n steps new sentences are generated by the callback')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for the optimizer (only applied on matmul weights)')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='Betas for the adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--log_dir', type=str, default='./logs', help='Sets logging directory for tensorboard logger.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    parser.add_argument('--num_workers', type=int, default=len(os.sched_getaffinity(0))-1, help='Num cpu workers used for training')
    parser.add_argument('--progress_bar', action='store_true', help=(
                            'Use a progress bar indicator for interactive experimentation. '
                            'Not to be used in conjuction with SLURM jobs'
                        ))
    parser.add_argument("--use_flash_attn", action="store_true", help="Use the Flash Attention module in the GPT model.")
    parser.add_argument("--precision", choices={"bf16", "bf16-mixed", "16-mixed", "16", "32"}, default="16-mixed")
    parser.add_argument("--compile", action="store_true", help="Compile the model for increased speed.")
    parser.add_argument("--pretrained_tokenizer", action="store_true", help="Use the pretrained tokenizer from OpenAI")
    args, _ = parser.parse_known_args()  # Parse known args and ignore the rest
    return args
