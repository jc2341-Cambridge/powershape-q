"""IonQ hardware entry point for the reduced signed-QAOA protocol."""

from .gate_model_qpu import main


if __name__ == "__main__":
    main("ionq")
