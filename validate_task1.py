#!/usr/bin/env python3
"""Simple validation script for Task 1.1 completion"""

print("🧪 Task 1.1 Final Validation")
print("=" * 40)

try:
    import src.core.config as cfg
    print("✅ Config module imported successfully")
    
    config_summary = cfg.Config.get_summary()
    print("📋 Configuration Summary:")
    for key, value in config_summary.items():
        print(f"  • {key}: {value}")
    
    validation_results = cfg.Config.validate_config()
    print("🔍 Validation Results:")
    for component, status in validation_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  • {component}: {status_icon}")
    
except Exception as e:
    print(f"❌ Config import failed: {e}")

try:
    import src.core.logger as logger
    print("✅ Logger module imported successfully")
    
    test_logger = logger.setup_logger("validation")
    test_logger.info("Logger validation successful!")
    
except Exception as e:
    print(f"❌ Logger import failed: {e}")

print("=" * 40)
print("🎉 Task 1.1 validation complete!")
