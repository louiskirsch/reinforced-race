import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import sensor.LightSensorArray;
import lejos.nxt.ColorSensor;
import lejos.nxt.Motor;
import lejos.nxt.SensorPort;
import lejos.nxt.comm.NXTConnection;
import lejos.nxt.comm.USB;
import lejos.nxt.comm.USBConnection;


public class BrickController {

	public static void main(String[] args) {
		//TODO: Right port
		LightSensorArray lsa = new LightSensorArray(SensorPort.S4);
		//TODO: Right port
		ColorSensor color = new ColorSensor(SensorPort.S1);
		
		NXTConnection connection = USB.waitForConnection();
		
		DataOutputStream dataOut = connection.openDataOutputStream();
		DataInputStream dataIn = connection.openDataInputStream();
		
		String commandString = null;
		while(true) {
			try {
				commandString = dataIn.readUTF();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			MotorCommand command = parseCommandString(commandString);
			command.getMotor().setSpeed(command.getSpeed());
			command.getMotor().forward();
		}
	}

	static MotorCommand parseCommandString(String commandString) {
		int indexMaxDataString = commandString.length() - 1;
		if (commandString.charAt(0) == '*'
				&& commandString.charAt(indexMaxDataString) == '#') {
			commandString = commandString.substring(1, indexMaxDataString);
			
			char motor = commandString.charAt(0);
			char direction = commandString.charAt(1);
			
			commandString = commandString.substring(2, indexMaxDataString-1);
			int speed = Integer.parseInt(commandString);
			
			return new MotorCommand(motor, direction, speed);
		} else {
			System.err.println("Error in received data");
			return null;
		}
		
		
	}

}
