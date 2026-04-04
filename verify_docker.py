import asyncio
import os
from schema_migration_env import SchemaMigrationEnv

async def main():
    image_name = os.getenv("IMAGE_NAME", "driftfix-env")
    print(f"Testing Docker image: {image_name}")
    
    # Initialize the environment with the image name
    # This will trigger the container start
    env = await SchemaMigrationEnv.from_docker_image(image_name=image_name, port=8001)
    
    try:
        print("Starting environment...")
        reset_result = await env.reset(task_id="add_email_column")
        print(f"Reset successful. Observation: {reset_result.observation.message}")
        
        print("Sending step: SELECT 1")
        step_result = await env.step({"sql": "SELECT 1"})
        print(f"Step successful. Reward: {step_result.reward}")
        print(f"Observation: {step_result.observation.message}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
