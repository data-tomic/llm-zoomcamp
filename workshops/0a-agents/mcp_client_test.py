# mcp_client_test.py (Corrected Again)
import asyncio
from fastmcp import Client  # We only need to import Client

async def main():
    async with Client("weather_server.py") as mcp_client:
        print("Client connected. Getting list of tools...")
        
        # The logic is identical, we just removed the optional type hint
        tools = await mcp_client.list_tools()
        
        print("\n--- Available Tools ---")
        print(tools)
        print("-----------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
