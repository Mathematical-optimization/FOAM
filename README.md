# DryShampoo
This is the source code used for the 'DryShampoo' experiment.

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        epsilon=1e-10,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_decoupled_weight_decay=True,
        inv_root_override=2,
        exponent_multiplier=1,
        grafting_config=AdamGraftingConfig(beta2=args.beta2, epsilon=1e-8),
        distributed_config=distributed_config,
        use_protected_eigh=True,
        matrix_root_inv_threshold=0.0
    )

The 'matrix_root_inv_threshold' hyperparameter has been added to the existing Shampoo optimizer. 

Max epsilon has been added as a parser argument.

vit.py: Source code for training ViT + ImageNet using the Distributed Shampoo optimizer. 

resnet.py: Source code for training Resnet-50 + ImageNet using the Distributed Shampoo optimizer.

conformer.py: Source code for training conformer + Librispeech using the Distributed Shampoo optimizer.

run_training.sh: Shell script to run vit.py.  
