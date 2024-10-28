
#%%
from drivers import oscilloscope_3034 as Scope
from drivers import siggen_3102 as SigGen
import json
import importlib


class MechanicalTurk():
    def __init__(self):
      
        print("Initializing LeTurc")
        self.config_path = "documentation"
        self.devices_dict = self.master_handshake()
        print("Devices: ", self.devices_dict)
        self.scope = self.devices_dict['Goated Oscilloscope']
        # self.siggen = self.devices_dict['Old AFG']


      
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        return
   
    def master_handshake(self):
        """
        Initializes and configures devices based on the configuration file.
        This method reads the configuration from a JSON file located at `self.config_path/devices.json`.
        It then initializes and configures oscilloscopes and signal generators based on the configuration.

        For each device, if it is marked as 'active' in the configuration, the corresponding module is imported
        and the device is initialized with its address.
        Returns:
            dict: A dictionary containing the initialized devices.
        """
        print("Master Handshaking")
        self.config_dict = json.load(open(f"{self.config_path}/equipment.json"))
        
        self.devices_dict = {}
        self.scopes = {}
        self.siggens = {}

        for key in self.config_dict:

            if key in ['Oscilloscope', 'oscilloscope', 'Scope', 'scope']:
                scopes_dict = self.config_dict[key]
                for maker_key in scopes_dict:
                    maker_dict = scopes_dict[maker_key]
                    for scope_key in maker_dict:
                        scope_listing = maker_dict[scope_key]
                        if scope_listing['active']:
                            
                            module = importlib.import_module(scope_listing['module_path'])
                            class_method = getattr(module, scope_listing['class_name'])
                            obj_instance = class_method(scope_listing['address'])

                            self.scopes[scope_listing['identifier']] = obj_instance
                            self.devices_dict[scope_listing['identifier']] = obj_instance

            elif key in ['Signal Generator', 'signal generator', 'SigGen', 'siggen']:
                siggen_dict = self.config_dict[key]
                for maker_key in siggen_dict:
                    maker_dict = siggen_dict[maker_key]
                    for siggen_key in maker_dict:
                        siggen_listing = maker_dict[siggen_key]
                        if siggen_listing['active']:
                            module = importlib.import_module(siggen_listing['module_path'])
                            class_method = getattr(module, siggen_listing['class_name'])
                            obj_instance = class_method(siggen_listing['address'])
                            self.siggens[siggen_listing['identifier']] = obj_instance
                            self.devices_dict[siggen_listing['identifier']] = obj_instance
                            
        
        return self.devices_dict
    


if __name__ == "__main__":

    with MechanicalTurk() as mt:
        import code 
        code.interact(local=locals())