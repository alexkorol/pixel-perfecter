from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
import tempfile
import logging
from PIL import Image
import io

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import processing functions from web_main.py
from src.web_main import process_file_for_web, parse_args

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Get processing parameters from form
            method = request.form.get('method', 'watershed')
            auto_scale = request.form.get('auto_scale', 'true').lower() == 'true'
            rows = int(request.form.get('rows', 2))
            cols = int(request.form.get('cols', 2))
            num_colors = int(request.form.get('num_colors', 16))
            manual_scale = int(request.form.get('manual_scale', 8)) if not auto_scale else None
            blur_size = int(request.form.get('blur_size', 5)) # Get blur_size

            # Validate blur_size (must be odd or 0)
            if blur_size < 0 or (blur_size > 0 and blur_size % 2 == 0):
                 return jsonify({'error': 'Blur size must be an odd number >= 1, or 0 to disable.'}), 400

            # Create args object similar to CLI args
            args = parse_args().parse_args([])  # Empty args list to use defaults
            args.input_dir = app.config['UPLOAD_FOLDER']
            args.output_dir = app.config['UPLOAD_FOLDER']
            args.debug_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'debug')

            # Process the image
            result_image = process_file_for_web(
                input_path,
                args,
                method=method,
                auto_scale=auto_scale,
                manual_scale=manual_scale,
                rows=rows,
                cols=cols,
                num_colors=num_colors,
                blur_size=blur_size # Pass blur_size
            )

            # Save the result to a temporary file in memory
            img_io = io.BytesIO()
            result_image.save(img_io, 'PNG')
            img_io.seek(0)

            # Return the processed image
            return send_file(img_io, mimetype='image/png')

        except ValueError as ve: # Catch specific ValueErrors for user feedback
            logger.warning(f"Validation Error: {str(ve)}")
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return jsonify({'error': f"An unexpected server error occurred: {str(e)}"}), 500
        finally:
            # Clean up temporary file
            if 'input_path' in locals() and os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file {input_path}: {e}")


if __name__ == '__main__':
    app.run(debug=True)