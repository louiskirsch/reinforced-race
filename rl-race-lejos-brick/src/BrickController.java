import java.io.DataInputStream;
import java.io.DataOutputStream;

import lejos.nxt.comm.NXTConnection;
import lejos.nxt.comm.USB;
import lejos.nxt.comm.USBConnection;


public class BrickController {

	public static void main(String[] args) {
		NXTConnection connection = USB.waitForConnection();
		
		DataOutputStream dataOut = connection.openDataOutputStream();
		DataInputStream dataIn = connection.openDataInputStream();
		
		SensorReadings reading = new SensorReadings();
		
		reading.red = 5;
		reading.green = 5;
		reading.blue = 5;
		
		//object.write
	}

}
