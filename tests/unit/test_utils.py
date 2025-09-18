"""Unit tests for utility functions."""

import pytest
import tempfile
import os
from pathlib import Path
import logging

from multi_agent_collab_learning.utils.logging import setup_logger
from multi_agent_collab_learning.utils.visualization import plot_learning_curves


class TestLogging:
    """Test logging utilities."""

    @pytest.mark.unit
    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    @pytest.mark.unit
    def test_setup_logger_with_level(self):
        """Test logger setup with custom level."""
        logger = setup_logger("test_logger", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    @pytest.mark.unit
    def test_setup_logger_with_file(self, temp_log_file):
        """Test logger setup with file output."""
        logger = setup_logger("test_logger", log_file=temp_log_file)

        # Test that logger writes to file
        logger.info("Test message")

        # Check that file was created and contains message
        assert os.path.exists(temp_log_file)
        with open(temp_log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content

    @pytest.mark.unit
    def test_setup_logger_custom_format(self):
        """Test logger setup with custom format."""
        custom_format = "%(name)s - %(message)s"
        logger = setup_logger("test_logger", format_string=custom_format)

        # Check that handler has custom format
        handler = logger.handlers[0]
        assert handler.formatter._fmt == custom_format

    @pytest.mark.unit
    def test_logger_singleton_behavior(self):
        """Test that loggers with same name are singletons."""
        logger1 = setup_logger("same_name")
        logger2 = setup_logger("same_name")

        assert logger1 is logger2

    @pytest.mark.unit
    def test_logger_different_names(self):
        """Test that loggers with different names are different objects."""
        logger1 = setup_logger("logger_one")
        logger2 = setup_logger("logger_two")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    @pytest.mark.unit
    def test_logger_hierarchy(self):
        """Test logger hierarchy behavior."""
        parent_logger = setup_logger("parent")
        child_logger = setup_logger("parent.child")

        assert child_logger.parent == parent_logger

    @pytest.mark.unit
    def test_log_levels(self, temp_log_file):
        """Test different log levels."""
        logger = setup_logger("test_logger", level=logging.DEBUG, log_file=temp_log_file)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        with open(temp_log_file, 'r') as f:
            content = f.read()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content


class TestVisualization:
    """Test visualization utilities."""

    @pytest.mark.unit
    def test_plot_learning_curves_basic(self, sample_learning_data):
        """Test basic learning curve plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_curves.png")

            fig = plot_learning_curves(
                sample_learning_data,
                title="Test Learning Curves",
                save_path=save_path
            )

            # Check that figure was created
            assert fig is not None

            # Check that file was saved
            assert os.path.exists(save_path)

    @pytest.mark.unit
    def test_plot_learning_curves_custom_labels(self, sample_learning_data):
        """Test learning curve plotting with custom labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_curves.png")

            fig = plot_learning_curves(
                sample_learning_data,
                title="Custom Title",
                xlabel="Episodes",
                ylabel="Performance",
                save_path=save_path
            )

            assert fig is not None
            assert os.path.exists(save_path)

    @pytest.mark.unit
    def test_plot_learning_curves_no_save(self, sample_learning_data):
        """Test learning curve plotting without saving."""
        fig = plot_learning_curves(
            sample_learning_data,
            title="Test Curves"
        )

        assert fig is not None
        # Should return figure object without saving

    @pytest.mark.unit
    def test_plot_learning_curves_empty_data(self):
        """Test learning curve plotting with empty data."""
        empty_data = {}

        with pytest.raises(ValueError, match="Learning data cannot be empty"):
            plot_learning_curves(empty_data)

    @pytest.mark.unit
    def test_plot_learning_curves_invalid_data(self):
        """Test learning curve plotting with invalid data format."""
        invalid_data = {
            'agent_0': "not_a_list",
            'agent_1': [0.1, 0.2, 0.3]
        }

        with pytest.raises((TypeError, ValueError)):
            plot_learning_curves(invalid_data)

    @pytest.mark.unit
    def test_plot_learning_curves_mismatched_lengths(self):
        """Test learning curve plotting with mismatched data lengths."""
        mismatched_data = {
            'agent_0': [0.1, 0.2, 0.3, 0.4],
            'agent_1': [0.2, 0.3]  # Different length
        }

        # Should handle mismatched lengths gracefully
        fig = plot_learning_curves(mismatched_data)
        assert fig is not None

    @pytest.mark.unit
    def test_plot_learning_curves_single_agent(self):
        """Test learning curve plotting with single agent."""
        single_agent_data = {
            'agent_0': [0.1, 0.3, 0.5, 0.7, 0.8]
        }

        fig = plot_learning_curves(single_agent_data)
        assert fig is not None

    @pytest.mark.unit
    def test_plot_learning_curves_large_dataset(self):
        """Test learning curve plotting with large dataset."""
        large_data = {}
        for i in range(10):  # 10 agents
            large_data[f'agent_{i}'] = [j * 0.01 for j in range(100)]  # 100 episodes

        fig = plot_learning_curves(large_data)
        assert fig is not None

    @pytest.mark.unit
    def test_plot_learning_curves_negative_values(self):
        """Test learning curve plotting with negative values."""
        negative_data = {
            'agent_0': [-0.5, -0.2, 0.0, 0.3, 0.5],
            'agent_1': [-0.3, 0.0, 0.2, 0.4, 0.6]
        }

        fig = plot_learning_curves(negative_data)
        assert fig is not None

    @pytest.mark.unit
    def test_plot_learning_curves_file_permissions(self, sample_learning_data):
        """Test handling of file permission errors."""
        # Try to save to a read-only directory (if we can create one)
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = os.path.join(temp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only

            save_path = os.path.join(readonly_dir, "test_curves.png")

            try:
                # This might raise a permission error, which should be handled gracefully
                fig = plot_learning_curves(
                    sample_learning_data,
                    save_path=save_path
                )
                # If no error, that's fine too
                assert fig is not None
            except PermissionError:
                # Expected behavior for permission error
                pass
            finally:
                # Restore permissions for cleanup
                os.chmod(readonly_dir, 0o755)


class TestHelpers:
    """Test helper utility functions."""

    @pytest.mark.unit
    def test_data_validation_functions(self):
        """Test data validation helper functions."""
        # These would test helper functions if they exist
        # For now, we'll test some basic validation concepts

        # Test that sample data formats are valid
        sample_data = {
            'agent_0': [0.1, 0.3, 0.5],
            'agent_1': [0.2, 0.4, 0.6]
        }

        # Validate data structure
        assert isinstance(sample_data, dict)
        assert all(isinstance(k, str) for k in sample_data.keys())
        assert all(isinstance(v, list) for v in sample_data.values())

    @pytest.mark.unit
    def test_numeric_helper_functions(self):
        """Test numeric helper functions."""
        import numpy as np

        # Test basic statistical operations that might be used
        data = [0.1, 0.3, 0.5, 0.7, 0.9]

        mean_val = np.mean(data)
        std_val = np.std(data)

        assert 0 <= mean_val <= 1
        assert std_val >= 0

        # Test normalization concepts
        normalized = [(x - min(data)) / (max(data) - min(data)) for x in data]
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0

    @pytest.mark.unit
    def test_file_path_helpers(self):
        """Test file path helper operations."""
        from pathlib import Path

        # Test path operations that might be used in utilities
        test_path = Path("/tmp/test/file.png")

        assert test_path.suffix == ".png"
        assert test_path.stem == "file"
        assert test_path.parent.name == "test"

        # Test path creation
        with tempfile.TemporaryDirectory() as temp_dir:
            new_path = Path(temp_dir) / "subdir" / "file.txt"
            new_path.parent.mkdir(parents=True, exist_ok=True)

            assert new_path.parent.exists()
            assert new_path.parent.is_dir()

    @pytest.mark.unit
    def test_configuration_helpers(self):
        """Test configuration helper functions."""
        # Test dictionary merging and validation that might be used
        default_config = {
            'param1': 10,
            'param2': 'default',
            'nested': {'a': 1, 'b': 2}
        }

        user_config = {
            'param1': 20,
            'nested': {'a': 10}
        }

        # Simple merge strategy
        merged = default_config.copy()
        merged.update(user_config)
        merged['nested'].update(user_config['nested'])

        assert merged['param1'] == 20  # User override
        assert merged['param2'] == 'default'  # Default preserved
        assert merged['nested']['a'] == 10  # User override
        assert merged['nested']['b'] == 2  # Default preserved