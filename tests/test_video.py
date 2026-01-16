"""Test suite for video processing module."""

import os
import tempfile
import pytest


class TestProcessVideo:
    """Test process_video function."""

    def test_process_video_invalid_path(self):
        """Test process_video with non-existent file"""
        cv2 = pytest.importorskip("cv2")
        from npcpy.data.video import process_video

        embeddings, texts = process_video("/nonexistent/video.mp4", "test_table")

        # Should return empty lists on error
        assert embeddings == []
        assert texts == []

    def test_process_video_returns_lists(self):
        """Test process_video return types"""
        cv2 = pytest.importorskip("cv2")
        from npcpy.data.video import process_video

        # Even with invalid input, should return list types
        result = process_video("", "table")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestSummarizeVideoFile:
    """Test summarize_video_file function."""

    def test_summarize_video_file_nonexistent(self):
        """Test summarize with non-existent file"""
        from npcpy.data.video import summarize_video_file

        result = summarize_video_file("/nonexistent/video.mp4")

        assert isinstance(result, str)
        assert "Video file:" in result
        assert "video.mp4" in result

    def test_summarize_video_file_with_cv2(self):
        """Test summarize extracts metadata when cv2 available"""
        cv2 = pytest.importorskip("cv2")
        from npcpy.data.video import summarize_video_file

        # Create a minimal test video file using cv2
        temp_dir = tempfile.mkdtemp()
        try:
            import numpy as np
            video_path = os.path.join(temp_dir, "test.avi")

            # Create a small test video
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (64, 64))

            # Write a few frames
            for i in range(10):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[:, :] = [i * 25, i * 25, i * 25]
                out.write(frame)

            out.release()

            # Test summarization
            result = summarize_video_file(video_path)

            assert isinstance(result, str)
            assert "Video file:" in result
            assert "test.avi" in result
            # Should have metadata if cv2 worked
            if "64x64" in result:
                assert "fps" in result

        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_summarize_video_file_language_param(self):
        """Test summarize accepts language parameter"""
        from npcpy.data.video import summarize_video_file

        # Should not raise with language param
        result = summarize_video_file("/fake/video.mp4", language="en")
        assert isinstance(result, str)

    def test_summarize_video_file_max_audio_param(self):
        """Test summarize accepts max_audio_seconds parameter"""
        from npcpy.data.video import summarize_video_file

        # Should not raise with custom max_audio_seconds
        result = summarize_video_file("/fake/video.mp4", max_audio_seconds=300)
        assert isinstance(result, str)

    def test_summarize_returns_transcript_note(self):
        """Test summarize includes transcript status"""
        from npcpy.data.video import summarize_video_file

        result = summarize_video_file("/nonexistent/video.mp4")

        # Should mention transcript status (either has one or note about missing)
        assert "transcript" in result.lower() or "No transcript" in result


class TestVideoModuleImports:
    """Test video module imports correctly."""

    def test_module_imports(self):
        """Test video module can be imported"""
        from npcpy.data import video

        assert hasattr(video, "process_video")
        assert hasattr(video, "summarize_video_file")

    def test_cv2_optional(self):
        """Test cv2 is handled as optional"""
        # Module should import even if cv2 not available
        # (it imports cv2 inside functions, not at module level)
        from npcpy.data.video import summarize_video_file

        # Should not raise ImportError at import time
        assert callable(summarize_video_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
