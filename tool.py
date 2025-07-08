def build_sam_optimizer(model, config):
    name = config.optimizer
    lr = config.canonical_lr

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.adam_decay, eps=config.epsilon)
    elif name == "adamw":
        if hasattr(config, 'twins_lr_factor'):
            factor = config.twins_lr_factor
            print("[Decrease lr of pre-trained model by factor {}]".format(factor))
            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "feat_encoder" not in n and 'context_encoder' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if ("feat_encoder" in n or 'context_encoder' in n) and p.requires_grad],
                    "lr": lr*factor,
                },
            ]
            full = [n for n, _ in model.named_parameters()]
            return torch.optim.AdamW(param_dicts, lr=lr, weight_decay=config.adamw_decay, eps=config.epsilon)
        else:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.adamw_decay, eps=config.epsilon)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_sam_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
        }
    """
    # scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.scheduler
    lr = config.canonical_lr

    if name == 'OneCycleLR':
        # scheduler = OneCycleLR(optimizer, )
        if hasattr(config, 'twins_lr_factor'):
            factor = config.twins_lr_factor
            scheduler = OneCycleLR(optimizer, [lr, lr*factor], config.num_steps+100,
                pct_start=0.05, cycle_momentum=False, anneal_strategy=config.anneal_strategy)
        else:
            scheduler = OneCycleLR(optimizer, lr, config.num_steps+100,
                pct_start=0.05, cycle_momentum=False, anneal_strategy=config.anneal_strategy)
    # elif name == 'MultiStepLR':
    #     scheduler.update(
    #         {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    #elif name == 'CosineAnnealing':
    #    scheduler = CosineAnnealingLR(optimizer, config.num_steps+100)
    #     scheduler.update(
    #         {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    # elif name == 'ExponentialLR':
    #     scheduler.update(
    #         {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
