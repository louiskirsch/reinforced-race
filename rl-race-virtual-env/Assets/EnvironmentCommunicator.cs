using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System;
using System.IO;

public class EnvironmentCommunicator : MonoBehaviour {

	public int port = 2851;
	public const int maxImageSize = 256;

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
		public const int ResponseBufferSize = maxImageSize * maxImageSize + 16;
		public byte[] responseBuffer = new byte[ResponseBufferSize];
	}

	private int DeterminePort() {
		string[] args = System.Environment.GetCommandLineArgs();
		for(int i = 0; i < args.Length; i++) {
			if(args[i] == "--port" && i + 1 < args.Length)
				return Convert.ToInt32(args[i + 1]);
		}
		// or return default port
		return port;
	}

	void Start() {
		cameraSensor = GetComponentInChildren<CameraSensor>();
		carState = GetComponent<CarState>();
		carController = GetComponent<CarController>();

		IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Any, DeterminePort());

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
		}
	}

	private void SendSensorData(int imageWidth, int imageHeight) {
		imageWidth = Math.Min(imageWidth, maxImageSize);
		imageHeight = Math.Min(imageHeight, maxImageSize);
		cameraSensor.TakePicture(image => {
			int responseSize = 6 + imageWidth * imageHeight;
			BinaryWriter writer = new BinaryWriter(new MemoryStream(client.responseBuffer));
			writer.Write(carState.Disqualified);
			writer.Write(carState.Finished);
			// We can't transfer floating point values, so let's transmit velocity * 2^16
			int velocity = IPAddress.HostToNetworkOrder((int)(carController.GetVelocity() * 0xffff));
			writer.Write(velocity);
			carState.ResetState();
			WriteCameraImage(writer, image);
			client.socket.BeginSend(client.responseBuffer, 0, responseSize, 0, new AsyncCallback(SendCallback), client);
		}, imageWidth, imageHeight);
	}

	private void ApplyAction(int vertical, int horizontal) {
		carController.ApplyAction(vertical, horizontal);
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

	private void WriteCameraImage(BinaryWriter writer, Texture2D image) {
		for(int y = 0; y < image.height; y++) {
			for(int x = 0; x < image.width; x++) {
				Color pixel = image.GetPixel(x, y);
				byte grayscale = (byte)(pixel.grayscale * 255);
				writer.Write(grayscale);
			}
		}
	}
}