"""CLI to generate ESM-2 embeddings in-process for an organized dataset directory.

    python -m superwater.embed --data_dir data/my_dataset --out_dir data/my_dataset_embeddings
"""
from argparse import ArgumentParser

import torch

from superwater.esm_embeddings import embed_dataset


def main(argv=None):
    parser = ArgumentParser(description="Generate ESM-2 per-residue embeddings for an organized dataset.")
    parser.add_argument('--data_dir', required=True, help='Organized dataset dir (one subfolder per complex).')
    parser.add_argument('--out_dir', required=True, help='Where to write <name>_chain_<i>.pt embedding files.')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip complexes that already have <name>_chain_0.pt in --out_dir.')
    args = parser.parse_args(argv)

    use_cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    embed_dataset(args.data_dir, args.out_dir, device, skip_existing=args.skip_existing)


if __name__ == '__main__':
    main()
