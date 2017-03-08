using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraSensor : MonoBehaviour {

	private bool shouldTakePicture = false;

	private Texture2D picture;
	private RenderTexture renderTexture;

	public delegate void PictureCallback(Texture2D picture);

	public void TakePicture(PictureCallback callback, int width, int height) {
		if(renderTexture == null || renderTexture.width != width || renderTexture.height != height) {
			if(renderTexture != null) {
				DestroyTextureResources();
			}
			CreateTextureResources(width, height);
		}
		StartCoroutine(TakePictureRoutine(callback));
	}

	private IEnumerator TakePictureRoutine(PictureCallback callback) {
		shouldTakePicture = true;
		yield return 0;
		callback(picture);
	}

	private void DestroyTextureResources() {
		renderTexture.Release();
		Destroy(renderTexture);
		Destroy(picture);
	}

	private void CreateTextureResources(int width, int height) {
		Camera cam = GetComponent<Camera>();
		cam.enabled = false;

		renderTexture = new RenderTexture(width, height, 24);
		cam.targetTexture = renderTexture;
		picture = new Texture2D(width, height, TextureFormat.RGB24, false);
	}

	void LateUpdate() {
		if (shouldTakePicture) {
			Camera cam = GetComponent<Camera>();

			cam.Render();
			RenderTexture.active = renderTexture;
			picture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
			RenderTexture.active = null;

			shouldTakePicture = false;
		}
	}
}
