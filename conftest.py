"""Put code/ on the import path so the tests can find the modules."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
