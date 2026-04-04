import asyncio
import os
from schema_migration_env import SchemaMigrationEnv

async def main():
    image_name = os.getenv("IMAGE_NAME", "driftfix-env")
    print(f"Testing Docker image: {image_name}")
    
    # Initialize the environment with the image name
    # This will trigger the container start
    env = await SchemaMigrationEnv.from_docker_image(image_name=image_name, port=8008)
    
    try:
        print("Starting environment...")
        reset_result = await env.reset(task_id="add_missing_column")
        print(f"Reset successful. Task: {reset_result.observation.task_description}")
        
        from schema_migration_env import SchemaMigrationAction
        print("Sending step: SELECT 1")
        action = SchemaMigrationAction(sql="SELECT 1;", action_type="execute")
        step_result = await env.step(action)
        print(f"Step successful. Reward: {step_result.reward}")
        print(f"Done status: {step_result.done}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
