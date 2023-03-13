import utils
import config as c
from simulation.generate_data import DataGeneration
from simulation.test_network import Test
from simulation.manual_control import ManualControl
from simulation.inference import Inference


def main():
    # Parse arguments
    options = utils.get_sim_options()

    # Choose simulation
    if options.generate_data:
        sim = DataGeneration()

    elif options.test_network:
        sim = Test()

    elif options.manual_control:
        sim = ManualControl()

    elif options.ask_params:
        print('Choose task:')
        print('0 --> generate targets randomly')
        print('1 --> reach target and go back to HB')
        print('2 --> fixed position')
        task = input('Task: ')

        print('\nChoose context:')
        print('0 --> static target')
        print('1 --> dynamic target')
        context = input('Context: ')

        print('\nChoose movement policy:')
        print('0 --> immediate onset')
        print('1 --> delayed reaching')
        print('2 --> dynamic onset')
        phases = input('Policy: ')

        c.task = 'test' if task == '0' else 'all' if task == '1' else 'single'
        c.context = 'static' if context == '0' else 'dynamic'
        c.phases = 'immediate' if phases == '0' else 'fixed' \
            if phases == '1' else 'dynamic'

        sim = Inference()

    else:
        sim = Inference()

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
