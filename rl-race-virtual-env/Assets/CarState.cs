using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarState : MonoBehaviour {

	private class TransformState {
		public Vector3 position;
		public Quaternion rotation;
		
		public TransformState(Vector3 position, Quaternion rotation) {
			this.position = position;
			this.rotation = rotation;
		}
	}

	public bool Disqualified { get; private set; }
	public bool Finished { get; private set; }

	public uint transformHistoryLength = 7;
	public float transformHistorySaveInterval = 1f;

	private GameObject[] checkpoints;
	private int checkpointsPassed = 0;
	private TransformState[] transformHistory;
	private int historyIndex = 0;

	void Start () {
		checkpoints = GameObject.FindGameObjectsWithTag("Checkpoint");

		TransformState startState = new TransformState(transform.position, transform.rotation);
		InitializeTransformHistory(startState);
		InvokeRepeating("SaveTransform", 0f, transformHistorySaveInterval);
	}

	private void InitializeTransformHistory(TransformState state) {
		transformHistory = new TransformState[transformHistoryLength];
		for(int i = 0; i < transformHistoryLength; i++) {
			transformHistory[i] = state;
		}
	}

	public void ResetState() {
		Disqualified = false;
		Finished = false;
	}

	private void SaveTransform() {
		if(Disqualified)
			return;
		TransformState state = new TransformState(transform.position, transform.rotation);
		transformHistory[historyIndex] = state;
		historyIndex = (historyIndex + 1) % transformHistory.Length;
	}

	private void ResetVehicle() {
		CarController carController = GetComponent<CarController>();
		Rigidbody body = GetComponent<Rigidbody>();
		TransformState historicalState = transformHistory[historyIndex];

		// Reset history, throw away bad states that are close to disqualification
		InitializeTransformHistory(historicalState);

		carController.ApplyAction(0f, 0f);
		transform.position = historicalState.position;
		transform.rotation = historicalState.rotation;
		body.velocity = Vector3.zero;
		body.angularVelocity = Vector3.zero;
		carController.ApplyAction(1f, 0f);
	}

	private void ResetCheckpoints() {
		checkpointsPassed = 0;
		foreach(GameObject checkpoint in checkpoints)
			checkpoint.SetActive(true);
	}
	
	void OnTriggerEnter(Collider other) {
		switch(other.tag) {
		case "Disqualification":
			Disqualified = true;
			ResetCheckpoints();
			ResetVehicle();
			Debug.Log("Car was disqualified");
			break;
		case "Checkpoint":
			other.gameObject.SetActive(false);
			checkpointsPassed++;
			Debug.Log("Car passed checkpoint");
			break;
		case "FinishLine":
			if(checkpointsPassed == checkpoints.Length) {
				Finished = true;
				ResetCheckpoints();
				Debug.Log("Car passed finish line");
			}
			break;
		}
	}
}
