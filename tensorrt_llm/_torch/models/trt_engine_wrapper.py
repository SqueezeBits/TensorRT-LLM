import argparse

import torch
import tensorrt as trt


class TRTEngineWrapper:
    def __init__(self, engine):
        self.engine = engine
        self.context = self.engine.create_execution_context()
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise NotImplementedError("Multiple inputs or outputs are not supported yet.")

    @classmethod
    def load_engine(cls, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Deserialization of TRT engine failed")
        return cls(engine)

    def get_input_names(self):
        input_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
        return input_names

    def get_output_names(self):
        output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        return output_names

    def __call__(self, input_tensor: torch.Tensor):
        self.context.set_input_shape(self.input_names[0], input_tensor.shape)
        output_shape = self.engine.get_tensor_shape(self.output_names[0])
        output_tensor = torch.empty(input_tensor.shape[0], *output_shape[1:], dtype=input_tensor.dtype, device=input_tensor.device)
        bindings = [int(input_tensor.data_ptr()), int(output_tensor.data_ptr())]
        self.context.execute_v2(bindings=bindings)
        return output_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    trt_engine = TRTEngineWrapper.load_engine(args.engine_path)
    example_input_tensor = torch.randn(1, 3, 504, 504).cuda()
    output_tensor = trt_engine(example_input_tensor)
    print(f"Engine run successfully. Output tensor shape: {output_tensor.shape}")


if __name__ == "__main__":
    main()