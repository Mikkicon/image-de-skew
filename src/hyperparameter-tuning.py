import torch
import optuna
import torch.nn.functional as F
from optuna.trial import TrialState, Trial

from image_util import BATCHSIZE, DEVICE, EPOCHS, IMAGE_SIZE, N_NN_OUTPUT_CLASSES, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES, get_train_test_dataset


def define_model(trial: Trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    conv_kernel_3 = 3
    pool_kernel_2 = 2

    in_features = 1
    image_size = [IMAGE_SIZE[0], IMAGE_SIZE[1]]
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128, 2)
        layers.append(torch.nn.Conv2d(in_features, out_features, kernel_size=conv_kernel_3, stride=1, padding=(conv_kernel_3 - 1) // 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(kernel_size=pool_kernel_2, stride=pool_kernel_2))
        in_features = out_features
        image_size[0] //= pool_kernel_2
        image_size[1] //= pool_kernel_2

    layers.append(torch.nn.Flatten())
    layers.append(torch.nn.Linear(in_features * image_size[0] * image_size[1], N_NN_OUTPUT_CLASSES))
    layers.append(torch.nn.LogSoftmax(dim=1))

    return torch.nn.Sequential(*layers)

def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)
    print(model)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    
    train_loader, valid_loader = get_train_test_dataset()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            data, target = sample['data'], sample['target']
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(valid_loader):
                data, target = sample['data'], sample['target']
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))