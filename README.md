# DryShampoo
This is the source code used for the 'DryShampoo' experiment.

    optimizer = DistributedShampoo(
        params=model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        **epsilon_config,
        momentum=0.0,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        grafting_config=AdamGraftingConfig(beta2=args.adam_grafting_beta2, epsilon=args.grafting_epsilon),
        use_decoupled_weight_decay=True,
        inv_root_override=2,
        exponent_multiplier=1,
        distributed_config=distributed_config,
        preconditioner_dtype=torch.float32,
        matrix_root_inv_threshold=args.matrix_root_inv_threshold,
        max_epsilon=args.max_epsilon
    )

The 'matrix_root_inv_threshold' and 'max_epsilon' hyperparameters have been added to the existing Shampoo optimizer. 

It is recommended to set both hyperparameters ‘start-preconditioning-step’ and ‘precondition-frequency’ of the existing Shampoo optimizer to 1.

vit.py: Source code for training ViT + ImageNet using the Distributed Shampoo optimizer. 

resnet.py: Source code for training Resnet-50 + ImageNet using the Distributed Shampoo optimizer.

conformer.py: Source code for training conformer + Librispeech using the Distributed Shampoo optimizer.

run_training.sh: Shell script to run vit.py.  
