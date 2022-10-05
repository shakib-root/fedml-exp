from torch import nn

import fedml
from fedml import FedMLRunner
from fedml.simulation import SimulatorSingleProcess
from fedml.core import ClientTrainer, ServerAggregator, FedMLAlgorithmFlow


from fedml import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_SP,
)


from fedml.constants import FedML_FEDERATED_OPTIMIZER_FEDAVG
from fedavg_sam import FedAvgSAMAPI


class FedMLRunnerSAM(FedMLRunner):
    """Extends FedMLRunner to allow SAM optimizer

    Args:
        device (_type_): _description_
        dataset (_type_): _description_
        model (_type_): _description_
        client_trainer (ClientTrainer): _description_
        server_aggregator (ServerAggregator): _description_
        algorithm_flow (FedMLAlgorithmFlow): _description_
    """
    def __init__(
        self,
        args,
        device,
        dataset,
        model: nn.Module,
        client_trainer: ClientTrainer = None,
        server_aggregator: ServerAggregator = None,
        algorithm_flow: FedMLAlgorithmFlow = None,
    ):
        if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION:
            init_runner_func = self._init_simulation_runner

        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
            init_runner_func = self._init_cross_silo_runner

        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE:
            init_runner_func = self._init_cross_device_runner
        else:
            raise Exception("no such setting")

        self.runner = init_runner_func(
            args, device, dataset, model, client_trainer, server_aggregator
        )

    def _init_simulation_runner(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        print("======= Calling _init_simulation_runner from FedMLRunnerSAM =======")
        if hasattr(args, "backend") and args.backend == FEDML_SIMULATION_TYPE_SP:
            runner = SimulatorSingleProcessSAM(
                args, device, dataset, model, client_trainer, server_aggregator
            )
        else:
            raise Exception(f"not such backend {args.backend}")

        return runner


class SimulatorSingleProcessSAM(SimulatorSingleProcess):
    def __init__(self, args, device, dataset, model, client_trainer=None, server_aggregator=None):
        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            self.fl_trainer = FedAvgSAMAPI(args, device, dataset, model)
        else:
            raise Exception("Exception")


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunnerSAM(args, device, dataset, model)
    fedml_runner.run()
