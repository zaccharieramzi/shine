from mdeq_lib.debug.backward_forward_ratio import eval_ratio_fb_classifier

if __name__ == '__main__':
    base_params = dict(
        n_gpus=1,
        n_epochs=100,
        seed=42,
        restart_from=50,
        n_samples=10,
    )
    eval_ratio_fb_classifier(
        dataset='cifar', model_size='LARGE', **base_params
    )
