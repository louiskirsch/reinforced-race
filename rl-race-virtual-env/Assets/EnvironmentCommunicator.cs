using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System;
using System.IO;

public class EnvironmentCommunicator : MonoBehaviour {

	public int port = 2851;

	private CameraSensor cameraSensor;
	private CarController carController;
	private CarState carState;

	private Socket listener;
	private Client client;

	private const byte REQUEST_READ_SENSORS = 1;
	private const byte REQUEST_WRITE_ACTION = 2;

	private class Client {

		public Client(Socket socket) {
			this.socket = socket;
		}

		public void BeginReceive(AsyncCallback callback) {
			socket.BeginReceive(buffer, 0, BufferSize, 0, callback, this);
		}

		public Socket socket;
		public const int BufferSize = 1024;
		public int bytesLeft = 0;
		public byte[] buffer = new byte[BufferSize];
		public volatile bool requestPending;
	}

	void Start() {
		cameraSensor = GetComponentInChildren<CameraSensor>();
		carState = GetComponent<CarState>();
		carController = GetComponent<CarController>();

		IPHostEntry ipHostInfo = Dns.GetHostEntry("");
		IPAddress ipAddress = ipHostInfo.AddressList[0];
		IPEndPoint localEndPoint = new IPEndPoint(ipAddress, port);

		listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

		try {
			listener.Bind(localEndPoint);
			listener.Listen(1);

			BeginAccept();
		} catch (Exception e) {
			Debug.LogException(e);
		}
	}

	private void BeginAccept() {
		listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);
	}

	private void AcceptCallback(IAsyncResult ar) {
		Socket listener = (Socket) ar.AsyncState;
		Socket handler = listener.EndAccept(ar);

		Debug.Log("Accepted new client");

		client = new Client(handler);
		client.BeginReceive(new AsyncCallback(ReadCallback));
	}

	private void ReadCallback(IAsyncResult ar) {  
		int bytesRead = client.socket.EndReceive(ar);  

		if (bytesRead > 0) {
			client.bytesLeft = bytesRead;
			client.requestPending = true;
			Debug.Log("Received new message from client");
		}
	}

	private void SendSensorData(int imageWidth, int imageHeight) {
		cameraSensor.TakePicture(image => {
			byte[] responseBuffer = new byte[2 + imageWidth * imageHeight];
			BinaryWriter writer = new BinaryWriter(new MemoryStream(responseBuffer));
			writer.Write(carState.Disqualified);
			writer.Write(carState.Finished);
			carState.ResetState();
			writer.Write(PackCameraImage(image));
			client.socket.BeginSend(responseBuffer, 0, responseBuffer.Length, 0, new AsyncCallback(SendCallback), client);
		}, imageWidth, imageHeight);
		Debug.Log("Sending sensor data to client");
	}

	private void ApplyAction(int vertical, int horizontal) {
		carController.ApplyAction(vertical, horizontal);
		Debug.Log("Applying new action received from client");
	}

	void Update() {
		if(client != null && client.requestPending) {
			client.requestPending = false;
			BinaryReader reader = new BinaryReader(new MemoryStream(client.buffer));
			while(client.bytesLeft > 0) {
				byte instruction = reader.ReadByte();
				client.bytesLeft--;
				switch(instruction) {
				case REQUEST_READ_SENSORS:
					int width = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					int height = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					client.bytesLeft -= 8;
					SendSensorData(width, height);
					break;
				case REQUEST_WRITE_ACTION:
					int vertical = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					int horizontal = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					client.bytesLeft -= 8;
					ApplyAction(vertical, horizontal);
					break;
				}
			}
			// Listen for next message
			client.BeginReceive(new AsyncCallback(ReadCallback));
		}
	}

	private void SendCallback(IAsyncResult ar) {
		client.socket.EndSend(ar);
	}  

	private byte[] PackCameraImage(Texture2D image) {
		byte[] buffer = new byte[image.width * image.height];
		for(int y = 0; y < image.height; y++) {
			for(int x = 0; x < image.width; x++) {
				Color pixel = image.GetPixel(x, y);
				byte grayscale = (byte)(pixel.grayscale * 255);
				buffer[y * image.width + x] = grayscale;
			}
		}
		return buffer;
	}
}