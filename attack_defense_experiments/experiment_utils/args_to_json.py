def args_to_json(args):
    assert args.local_optimizer in ['sgd', 'adam']
    json_config = {
        "trainer": {
            "_base_": "base_sync_trainer",
            # there are different types of aggregator
            # fed avg doesn't require lr, while others such as fed_avg_with_lr or fed_adam do
            "_base_": "base_sync_trainer",
            "server": {
                "_base_": "base_sync_server",
                "server_optimizer": {
                    "_base_": "base_fed_avg_with_lr",
                    "lr": args.lr,
                    "momentum": args.momentum,
                },
                # type of user selection sampling
                "active_user_selector": {"_base_": "base_uniformly_random_active_user_selector"},
            },
            "client": {
                # number of client's local epoch
                "epochs": args.local_epochs,
                "optimizer": {
                    "_base_": "base_optimizer_sgd",
                    # client's local learning rate
                    "lr": args.local_lr,
                    # client's local momentum
                    "momentum": 0,
                } if args.local_optimizer == 'sgd' else {
                    "_base_": "base_optimizer_adam",
                    # client's local learning rate
                    "lr": args.local_lr,
                },
            },
            # number of users per round for aggregation
            "users_per_round": args.num_cl,
            # total number of global epochs
            # total #rounds = ceil(total_users / users_per_round) * epochs
            "epochs": args.epochs,
            # frequency of reporting train metrics
            "train_metrics_reported_per_epoch": 100,
            # frequency of evaluation per epoch
            "eval_epoch_frequency": 1,
            "do_eval": True,
            # should we report train metrics after global aggregation
            "report_train_metrics_after_aggregation": True,
        }
    }
    return json_config
