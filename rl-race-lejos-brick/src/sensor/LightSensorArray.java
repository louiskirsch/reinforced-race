package sensor;

import lejos.nxt.I2CPort;
import lejos.nxt.I2CSensor;

/**
 * @author MisterErwin, skoehler and Sebastian Bischoff (sebastian@salzreute.de)
 *         {@link http://lejos.sourceforge.net/forum/viewtopic.php?f=7&t=4312}
 */
public class LightSensorArray extends I2CSensor {

	private byte[] buf = new byte[8];

	private final static byte COMMAND = 0x41;

	public enum Command {
		CALIBRATE_WHITE('W'), CALIBRATE_BLACK('B'), SLEEP('D'), WAKEUP('P'), FREQ_60HZ(
				'A'), FREQ_50HZ('E'), FREQ_UNIVERSAL('U');

		private Command(char value) {
			this.value = value;

		}

		final char value;
	}

	/**
	 * 
	 * @param port
	 *            The port where the sensor is plugged, for example
	 *            "SensorPort.S1"
	 * @param address
	 *            The I2C bus address of LightSensorArray
	 */

	public LightSensorArray(I2CPort port, int address) {
		super(port, address, I2CPort.LEGO_MODE, TYPE_LOWSPEED_9V);
	}

	/**
	 * 
	 * @param port
	 *            The port where the sensor is plugged, like "SensorPort.S1"
	 */

	public LightSensorArray(I2CPort port) {
		this(port, 0x14 /* DEFAULT_I2C_ADDRESS */);
	}

	/**
	 * 
	 * @param cmd
	 *            The command which is sent to the LightSensorArray.
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */

	public int sendCommand(Command cmd) {
		return sendData(COMMAND, (byte) cmd.value);
	}

	/**
	 * Sets an new I2C-address for the LightSensorArray
	 * 
	 * @param newAddress
	 *            The new I2C-address of the LightSensorArray
	 */
	public void newI2CAddress(byte newAddress) {
		this.sendData(COMMAND, (byte) 0xA0);
		this.sendData(COMMAND, (byte) 0xAA);
		this.sendData(COMMAND, (byte) 0xA5);
		this.sendData(COMMAND, newAddress);
	}

	/**
	 * Configure LightSensorArray for US region and other regions with 60 Hz
	 * electrical frequency.
	 * 
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */
	public int configureUS() {
		return this.sendCommand(Command.FREQ_60HZ);
	}

	/**
	 * Configure LightSensorArray for EU region and other regions with 50 Hz
	 * electrical frequency.
	 * 
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */
	public int configureEU() {
		return this.sendCommand(Command.FREQ_50HZ);
	}

	/**
	 * Configure LightSensorArray for universal frequency so you can use it
	 * everywhere, this is the default mode.
	 * 
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */
	public int configureUniversal() {
		return this.sendCommand(Command.FREQ_UNIVERSAL);
	}

	/**
	 * Wakes up the LightSensorArray. The sensor wakes up on its own when any
	 * activity begins.
	 * 
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */
	public int wakeUp() {
		return this.sendCommand(Command.WAKEUP);
	}

	/**
	 * Put the LightSensorArray to sleep. After 1 minute of inactivity the
	 * sensor goes automatically to sleep. "Sleep"-mode conserves power (Current
	 * Consumption: 3.0 mA in "Sleep"-mode, while awake max 5.7 mA).
	 * 
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */
	public int sleep() {
		return this.sendCommand(Command.SLEEP);
	}

	/**
	 * Calibrates black (All sensors should be on a black surface)
	 * 
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */

	public int calibrateBlack() {
		return this.sendCommand(Command.CALIBRATE_BLACK);
	}

	/**
	 * Calibrates white (All sensors should be on a white surface)
	 * 
	 * @return Returns 0 when no error occured and a negative number when an
	 *         error occured.
	 */

	public int calibrateWhite() {
		return this.sendCommand(Command.CALIBRATE_WHITE);
	}

	/**
	 * Gets a value from one Light Sensor
	 * 
	 * @param slot
	 *            The slot on the Light Sensor Array
	 * @return the Light Value
	 * @throws IllegalArgumentException
	 *             If slot is not between 0 and 7
	 */
	public int getLightValue(int slot) {
		if (slot > 7 || slot < 0) {
			throw new IllegalArgumentException("Slot MUST be between 0 and 7!");
		}

		int b = getData(0x42, buf, 1);

		return (b == 0 ? (buf[0] & 0xff) : -1);
	}

	/**
	 * Gets the values from the LightSensorArray
	 * 
	 * @return Array of ints
	 */
	public int[] getLightValues() {
		int[] ret = new int[8];

		int err = getData(0x42, buf, 8);
		if (err == 0) {
			for (int i = 0; i < 8; i++)
				ret[i] = buf[i] & 0xff;
		} else {
			for (int i = 0; i < 8; i++)
				ret[i] = -1;
		}

		return ret;
	}

}