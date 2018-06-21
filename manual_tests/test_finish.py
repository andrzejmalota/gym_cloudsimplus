from py4j.java_gateway import JavaGateway

gateway = JavaGateway()
simulation_environment = gateway.entry_point

simulation_environment.reset()
print("Starting a simulation")
simulation_environment.step(0)
for i in range(50):
    simulation_environment.step(1)
    result = simulation_environment.render()
    print("Added a VM: " + str(gateway.jvm.java.util.Arrays.toString(result[0][-5:])))

done = False

while not done:
    result = simulation_environment.step(0)
    done = result.isDone()
    state = simulation_environment.render()
    print("Did nothing: " + str(gateway.jvm.java.util.Arrays.toString(state[0][-5:])))
    print("Result: " + str(done))

print("End of simulation")
