using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarState : MonoBehaviour {

	private Vector3 initialPosition;
	private Quaternion initialRotation;

	public bool Disqualified { get; private set; }
	public bool Finished { get; private set; }

	private GameObject[] checkpoints;
	private int checkpointsPassed = 0;

	void Start () {
		initialPosition = transform.position;
		initialRotation = transform.rotation;
		checkpoints = GameObject.FindGameObjectsWithTag("Checkpoint");
	}

	public void ResetState() {
		Disqualified = false;
		Finished = false;
	}

	private void ResetVehicle() {
		transform.position = initialPosition;
		transform.rotation = initialRotation;
		Rigidbody body = GetComponent<Rigidbody>();
		body.velocity = Vector3.zero;
		body.angularVelocity = Vector3.zero;
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
