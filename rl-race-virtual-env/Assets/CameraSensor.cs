using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraSensor : MonoBehaviour {

	private bool shouldTakePicture = false;
	private int resWidth;
	private int resHeight;
	private Texture2D lastPicture;

	public delegate void PictureCallback(Texture2D picture);

	public void TakePicture(PictureCallback callback, int width, int height) {
		resWidth = width;
		resHeight = height;
		StartCoroutine(TakePictureRoutine(callback));
	}

	private IEnumerator TakePictureRoutine(PictureCallback callback) {
		shouldTakePicture = true;
		yield return 0;
		callback(lastPicture);
	}

	void LateUpdate() {
		if (shouldTakePicture) {
			Camera cam = GetComponent<Camera> ();

			RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
			cam.targetTexture = rt;
			Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
			cam.Render();
			RenderTexture.active = rt;
			screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
			cam.targetTexture = null;
			RenderTexture.active = null; // JC: added to avoid errors
			Destroy(rt);

			lastPicture = screenShot;
			
			shouldTakePicture = false;
		}
	}
}
