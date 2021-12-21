from collections import defaultdict
import numpy as np
import torch

from main import (
    enable_determinism,
    generate_argument_parser,
    init_text_logger,
    init_dataset,
    test_one_epoch,
)
from model import DCP
from util import npmat2euler


def test(args, net, test_loader, textio):

    (
        test_loss,
        test_cycle_loss,
        test_mse_ab,
        test_mae_ab,
        test_mse_ba,
        test_mae_ba,
        test_rotations_ab,
        test_translations_ab,
        test_rotations_ab_pred,
        test_translations_ab_pred,
        test_rotations_ba,
        test_translations_ba,
        test_rotations_ba_pred,
        test_translations_ba_pred,
        test_eulers_ab,
        test_eulers_ba,
    ) = test_one_epoch(args, net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.nanmean(
        (test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2
    )
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.nanmean(
        np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab))
    )
    test_t_mse_ab = np.nanmean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.nanmean(np.abs(test_translations_ab - test_translations_ab_pred))

    textio.cprint("==FINAL TEST==")
    textio.cprint("A--------->B")
    textio.cprint(
        "EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, "
        "rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f"
        % (
            -1,
            test_loss,
            test_cycle_loss,
            test_mse_ab,
            test_rmse_ab,
            test_mae_ab,
            test_r_mse_ab,
            test_r_rmse_ab,
            test_r_mae_ab,
            test_t_mse_ab,
            test_t_rmse_ab,
            test_t_mae_ab,
        )
    )
    return dict(
        mse=test_mse_ab,
        rmse=test_mse_ab,
        mae=test_mae_ab,
        r_mse=test_r_mse_ab,
        r_rmse=test_r_rmse_ab,
        r_mae=test_r_mae_ab,
        t_mse=test_t_mse_ab,
        t_rmse=test_t_rmse_ab,
        t_mae=test_t_mae_ab,
    )


def main():
    # Parser
    parser = generate_argument_parser()
    parser.add_argument(
        "--init_model_list", nargs="+", help="List of models to compute metrics for."
    )
    args = parser.parse_args()

    enable_determinism(args.seed)

    textio = init_text_logger(args)
    test_loader = init_dataset(args, split="test")
    net = DCP(args).cuda()

    data = defaultdict(list)
    for i, model in enumerate(sorted(args.init_model_list)):
        print(f"Run: {i} Model: {model}")
        net.load_state_dict(torch.load(model), strict=True)
        run = test(args, net, test_loader, textio)
        for k, v in run.items():
            data[k].append(v)

    for k, v in data.items():
        print(f"{k}:\t{np.nanmean(v):>9.6f} (mean)\t{np.nanstd(v):0.3e} (std)")


if __name__ == "__main__":
    main()
