import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import serial.tools

class DeepBlue():
    def __init__(self, address):
        self.rm = pyvisa.ResourceManager()
        self.client = self.rm.open_resource(address)
        print("Connected to: ", self.identity())


    def __enter__(self):
        return self
    
    def write(self, cmd):
        self.client.write(cmd)
        return 

    def query(self, cmd):
        return self.client.query(cmd)

    def identity(self):
        return self.query("*IDN?")

###############################################################

    ### in Hertz
    def get_frequency(self, channel):
        ans = self.query(f"SOUR{channel}:FREQ:FIXED?")
        try:
            ans = float(ans[:-1])
            return ans
        except:
            print("get_frequency : Invalid reply from instrument!")
            print(ans)
            return -1.0
    
    ### in Volt
    def get_amplitude(self, channel):
        ans = self.query(f"SOURce{channel}:AM:DEPTh?")
        try:
            ans = float(ans[:-1])
            return ans
        except:
            print("get_amplitude : Invalid reply from instrument")
            print(ans)
            return -1.0
    
###############################################################

    def set_frequency(self, channel, frequency):
        self.write(f"SOURCE{channel}:FREQUENCY {frequency}")
        return


    def set_amplitude(self,channel, amplitude):
        self.write(f"SOUR{channel}:AM {amplitude}")
        return
    


    def waveform(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        return





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

    with DeepBlue(address) as db:
        import code 
        code.interact(local=locals())
