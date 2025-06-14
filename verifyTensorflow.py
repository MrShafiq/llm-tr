import tensorflow as tf
import time

print("=" * 60)
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("✅ GPU(s) detected:")
    for gpu in gpus:
        print("  -", gpu)
else:
    print("❌ No GPU detected.")

print("=" * 60)

# Optional: Perform a matrix multiplication test on GPU
if gpus:
    try:
        with tf.device('/GPU:0'):
            print("🔁 Running matrix multiplication on GPU...")
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start = time.time()
            c = tf.matmul(a, b)
            tf.print("✅ Done. Elapsed time:", time.time() - start, "seconds")
    except RuntimeError as e:
        print("⚠️ Runtime error:", e)
else:
    print("💡 Tip: If you recently installed TensorFlow, reboot may help.")
