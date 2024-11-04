import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import serial.tools

class AlphaZero:
    ########################################################################
    def __init__(self, address):
        self.rm = pyvisa.ResourceManager()
        self.client = self.rm.open_resource(address)
        print("Connected to: ", self.identity())


        self.data = np.array([])



    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        return
     
    def write(self, cmd):
        self.client.write(cmd)
        return 

    def query(self, cmd):
        return self.client.query(cmd)[:-1]

    def identity(self):
        return self.query("*IDN?")
    
    def exit(self):
        return
##########################################################################
    def set_source_channel(self, channel):
        self.write(f"DAT:SOU CH{channel}")
        return
    
    def get_source_channel(self):
        return self.query("DAT:SOU?")

    def set_encoding(self, encoding):
        self.write(f"DAT:ENC {encoding}")
        return

    def get_encoding(self):
        return self.query("DAT:ENC?")

    ############################################################################
    def set_trigger_mode(self, mode, trigger = "A"):
        self.write(f"TRIG:{trigger}:MOD {mode}")
        return
    
    def get_trigger_mode(self, trigger = "A"):
        return self.query(f"TRIG:{trigger}:MOD?")

    def set_trigger_level(self, channel, level, trigger = "A"):
        self.write(f"TRIG:{trigger}:LEV:CH{channel} {level}")
        return
    
    def get_trigger_level(self, channel, trigger = "A"):
        return self.query(f"TRIG:{trigger}:LEV:CH{channel}?")
    
    def set_trigger_coupling(self, coupling, mode = "Edge", trigger = "A"):
        self.write(f"TRIG:{trigger}:{mode}:COU {coupling}")
        return
        
    def get_trigger_coupling(self, mode = "Edge", trigger = "A"):
        return self.query(f"TRIG:{trigger}:{mode}:COU?")
    
    def set_trigger_source (self, source, mode = "Edge", trigger = "A"):
        self.write(f"TRIG:{trigger}:{mode}:SOU {source}")
        return

    def get_trigger_source (self, mode = "Edge", trigger = "A"):
        return self.query(f"TRIG:{trigger}:{mode}:SOU?")
    
############################################################################

    def set_voltage_division(self, division, channel):
        self.write(f"CH{channel}:SCA {division}")
        return 
    
    def get_voltage_division(self, channel):
        return self.query(f"CH{channel}:SCA?")
    
    def set_voltage_coupling(self, coupling, channel):
        self.write(f"CH{channel}:COU {coupling}")
        return
    
    def get_voltage_coupling(self, channel):
        return self.query(f"CH{channel}:COU?")
    
    def set_voltage_offset(self, offset, channel):
        self.write(f"CH{channel}:OFFS {offset}")
        return
    
    def get_voltage_offset(self, channel):
        return self.query(f"CH{channel}:OFFS?")
    
#########################################################################

    def set_time_division(self, division):
        self.write(f"TIM:SCA {division}")
        return
    
    def get_time_division(self):
        return self.query("TIM:SCA?")
    
    def set_time_offset(self, offset):
        self.write(f"TIM:OFFS {offset}")
        return
    
    def get_time_offset(self):
        return self.query("TIM:OFFS?")

#########################################################################

    def get_trace(self, channel, points):
        self.set_source_channel(channel)
        self.set_encoding("ASCII")
        self.write("DAT:WID 1")
        self.write(f"DAT:STAR 1")
        self.write(f"DAT:STOP {points}")
        trace = self.query("CURV?")
        trace = np.array(trace.split(",")).astype(np.int64) #### because we expect negative values too

        return trace


if __name__ == "__main__":
    rm = pyvisa.ResourceManager()
    addresses = rm.list_resources()
    print("Please choose the address of the device")
    for i, address in enumerate(addresses):
        print(f"{i}: {address}")
    print("Your choice:")

    choice = int(input())
    address = addresses[choice]
    print("You chose: ", address)        

    with AlphaZero(address) as az:
        import code 
        code.interact(local=locals())
