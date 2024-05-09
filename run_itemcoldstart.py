import argparse

from itemcoldstart.quick_start import run_itemcoldstart, run_itemcoldstarts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="Firzen", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Amazon_Beauty", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    if args.nproc == 1 and args.world_size <= 0:
        run_itemcoldstart(
            model=args.model, dataset=args.dataset, config_file_list=config_file_list
        )
    else:
        if args.world_size == -1:
            args.world_size = args.nproc
        import torch.multiprocessing as mp

        mp.spawn(
            run_itemcoldstarts,
            args=(
                args.model,
                args.dataset,
                config_file_list,
                args.ip,
                args.port,
                args.world_size,
                args.nproc,
                args.group_offset,
            ),
            nprocs=args.nproc,
        )
