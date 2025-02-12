class Noise:
    def __init__(self, noise_params):
        '''Initialize noise with noise parameters
        Args:
            noise_params: dict, noise parameters, like {theta_1: 0.5}
        '''
        self.class_name = "Noise"
        self.required_keys = {}
        self.allowed_keys = {}
        self.check_noise_params(noise_params)
        
    
    def check_noise_params(self, noise_params):
        '''Check if noise parameters are valid for noise class
        Args:
            noise_params: dict, noise parameters
        Returns:
            bool, True if noise parameters are valid
        '''
        for key in self.required_keys:
            if key not in noise_params:
                raise ValueError(f"Missing noise parameter for {self.class_name}: {key}")
        for key in noise_params:
            if key not in self.required_keys and key not in self.allowed_keys:
                print(f"WARNING: Invalid parameter for {self.class_name}: {key}")
        
    def apply_noise(self, input):
        '''Apply noise to input
        Args:
            input: str, input text
        Returns:
            str, noised text
        '''
        raise NotImplementedError

    def record_noiser_artifacts(self):
        '''Save noiser artifacts to output file
        '''
        pass
    