import asyncio
from schema_migration_env import SchemaMigrationEnv, SchemaMigrationAction

async def test():
    # 1. Reset the environment (starts a local task on port 7860)
    env = SchemaMigrationEnv(base_url="http://localhost:7860")
    result = await env.reset(task_id="add_missing_column")
    
    # Calculate passing queries from the list of results
    passing = len([qr for qr in result.observation.query_results if qr.passed])
    total = len(result.observation.query_results)
    
    print(f"Task Started: {result.observation.task_id}")
    print(f"Initial Queries: {passing} / {total} passing")

    # 2. Take a dummy action
    print("\nExecuting dummy action (SELECT 1;)...")
    action = SchemaMigrationAction(sql="SELECT 1;", action_type="execute")
    step = await env.step(action)
    print(f"Step Reward: {step.reward:.2f}")
    
    # 3. Take a valid action (adding the missing column)
    print("\nExecuting fix (ALTER TABLE employees ADD COLUMN salary...)...")
    action = SchemaMigrationAction(
        sql="ALTER TABLE employees ADD COLUMN salary INTEGER DEFAULT 60000;",
        action_type="execute"
    )
    step = await env.step(action)
    
    # Recalculate passing queries after the step
    passing = len([qr for qr in step.observation.query_results if qr.passed])
    
    print(f"Correct Action Reward: {step.reward:.2f}")
    print(f"Status: {'Done' if step.done else 'In Progress'}")
    print(f"Queries Passing: {passing} / {total}")

    await env.close()

if __name__ == "__main__":
    asyncio.run(test())
