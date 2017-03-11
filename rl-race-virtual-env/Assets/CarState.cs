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

	public uint maxTransformHistoryLength = 15;
	public float transformHistorySaveDistance = 1f;
	public float resetMinimumDistance = 5f;

	private GameObject[] checkpoints;
	private int checkpointsPassed = 0;

	private LinkedList<TransformState> transformHistory;

	private Vector3 lastPosition;
	private float distanceDriven = 0f;
	private float lastHistoryTransformDistance = 0f;

	void Start () {
		checkpoints = GameObject.FindGameObjectsWithTag("Checkpoint");

		TransformState startState = new TransformState(transform.position, transform.rotation);
		InitializeTransformHistory(startState);

		lastPosition = transform.position;
	}

	private void InitializeTransformHistory(TransformState state) {
		transformHistory = new LinkedList<TransformState>();
		transformHistory.AddLast(state);
	}

	public void ResetState() {
		Disqualified = false;
		Finished = false;
	}

	private void SaveTransform() {
		if(Disqualified)
			return;
		TransformState state = new TransformState(transform.position, transform.rotation);
		transformHistory.AddLast(state);
		if(transformHistory.Count > maxTransformHistoryLength)
			transformHistory.RemoveFirst();
		lastHistoryTransformDistance = distanceDriven;
	}

	void Update() {
		distanceDriven += (transform.position - lastPosition).magnitude;
		lastPosition = transform.position;
		if(distanceDriven - lastHistoryTransformDistance > transformHistorySaveDistance)
			SaveTransform();
	}

	private TransformState SampleHistoryState() {
		int prohibitedHistory = (int)(resetMinimumDistance / transformHistorySaveDistance);
		int historySample = Random.Range(0, System.Math.Max(1, transformHistory.Count - prohibitedHistory));

		// Delete all following history states
		while(transformHistory.Count - 1 != historySample)
			transformHistory.RemoveLast();
		
		return transformHistory.Last.Value;
	}

	private void ResetVehicle() {
		CarController carController = GetComponent<CarController>();
		Rigidbody body = GetComponent<Rigidbody>();
		TransformState historicalState = SampleHistoryState();

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
