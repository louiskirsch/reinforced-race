using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ScreenshotTaker : MonoBehaviour {

	public int resWidth = 64; 
	public int resHeight = 64;

	private static string ScreenShotDirectory() {
		return string.Format("{0}/rl-race-screenshots", 
			System.Environment.GetFolderPath(System.Environment.SpecialFolder.MyPictures));
	}

	private static string ScreenShotName(int width, int height) {
		return string.Format("{0}/screen_{1}x{2}_{3}.png", 
			ScreenShotDirectory(), 
			width, height, 
			System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
	}

	void LateUpdate() {
		bool takeShot = Input.GetKeyDown("k");
		if (takeShot) {
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
			byte[] bytes = screenShot.EncodeToPNG();
			string filename = ScreenShotName(resWidth, resHeight);
			Directory.CreateDirectory(ScreenShotDirectory());
			System.IO.File.WriteAllBytes(filename, bytes);
			Debug.Log(string.Format("Took screenshot to: {0}", filename));
		}
	}
}
