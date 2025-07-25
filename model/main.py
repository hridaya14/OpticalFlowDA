import argparse
import os

from core.memflow.core.utils.misc import process_cfg

from .model import Model
def main_worker(rank, cfg):
    trainer = Model(cfg)
    trainer.train_memflow(rank, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # DDP
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
    parser.add_argument('--DDP', action='store_true', help='DDP')
    parser.add_argument('--eval_only', action='store_true', default=False, help='eval only')

    args = parser.parse_args()

    if args.stage == 'things':
        from configs.things_memflownet import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel_memflownet import get_cfg
    elif args.stage == 'spring_only':
        from configs.spring_memflownet import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti_memflownet import get_cfg

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    if args.DDP:
        args.world_size = args.nodes * args.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22323'
    else:
        args.world_size = 1

    cfg = get_cfg() # type: ignore
    cfg.update(vars(args))
    process_cfg(cfg)
    if not cfg.eval_only:
        loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    model = Model(cfg)

    print("  <====== Starting Training ====>")
    mp.spawn(main_worker, nprocs=args.world_size, args=(cfg,))

    print("  <====== Training Finished ====>")


