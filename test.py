import testing.task3.QuanTest1 as q1
import testing.task3.QuanTest2 as q2
import testing.task4.Test_Cases.signalcompare as cmp1
import testing.task4.Remove_DC_component.CompareSignals as cmp2
from utils import dft, quantization, read_signal, remove_dc_component

signal1 = read_signal("signals/task4/input_Signal_DFT.txt")
result1 = dft(signal1, False)
output1 = read_signal("testing/task4/Test_Cases/DFT/Output_Signal_DFT_A,Phase.txt")

signal2 = read_signal("signals/task4/Input_Signal_IDFT_A,Phase.txt")
result2 = dft(signal2, True)
output2 = read_signal("testing/task4/Test_Cases/IDFT/Output_Signal_IDFT.txt")

result1_a, result1_p = result1.split()
output1_a, output1_p = output1.split()

result2_t, result2_a = result2.split()
output2_t, output2_a = output2.split()

test1_amp = cmp1.SignalComapreAmplitude(result1_a, output1_a)
test1_phase = cmp1.SignalComaprePhaseShift(result1_p, output1_p)
print(f"Task 4 DFT {test1_amp} {test1_phase}")

test2_amp = cmp1.SignalComapreAmplitude(result2_a, result2_a)
print(f"Task 4 IDFT {test2_amp}")

signal3 = read_signal("testing/task4/Remove_DC_component/DC_component_input.txt")
freq_signal3 = dft(signal3)
freq_signal3 = remove_dc_component(freq_signal3)
result3 = dft(freq_signal3, True)
result3_t, result3_a = result3.split()
test3 = cmp2.SignalsAreEqual(
    "Task 4 Remove DC Component",
    "testing/task4/Remove_DC_component/DC_component_output.txt",
    result3_t,
    result3_a,
)
