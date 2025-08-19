print("Running all tests...")
# You can expand this to call validate.py for all ops
import subprocess
result = subprocess.run(["python", "../python/validate.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("❌ Some tests failed")
else:
    print("✅ All tests passed")
