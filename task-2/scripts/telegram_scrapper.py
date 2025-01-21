from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

# Strip and load credentials
try:
    api_id = os.getenv('TG_API_ID')
    api_hash = os.getenv('TG_API_HASH')
    phone = os.getenv('phone')

    # Debug: Print the raw values to ensure they are loaded correctly
    print(f"Loaded values from .env: TG_API_ID={repr(api_id)}, TG_API_HASH={repr(api_hash)}, phone={repr(phone)}")

    # Convert API ID to an integer
    api_id = int(api_id.strip())
    api_hash = api_hash.strip()
    phone = phone.strip()

except (ValueError, AttributeError):
    print("Error: Check your .env file for correctly formatted values.")
    exit(1)


# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title  # Extract the channel's title
        async for message in client.iter_messages(entity, limit=10000):
            media_path = None
            if message.media and hasattr(message.media, 'photo'):
                # Create a unique filename for the photo
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                # Download the media to the specified directory if it's a photo
                await client.download_media(message.media, media_path)

            # Write the channel title along with other data
            writer.writerow([
                channel_title,
                channel_username,
                message.id,
                message.message or "",
                message.date,
                media_path or "None"
            ])
    except Exception as e:
        print(f"Error scraping channel {channel_username}: {e}")

# Main function to scrape multiple channels
async def main():
    # Initialize Telegram client
    client = TelegramClient('scraping_session', api_id, api_hash)

    await client.start(phone=phone)

    # Create a directory for media files
    media_dir = 'photos'
    os.makedirs(media_dir, exist_ok=True)

    # Open the CSV file and prepare the writer
    csv_file = 'telegram_data.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'Message ID', 'Message', 'Date', 'Media Path'])  # Header row

        # List of Telegram channels to scrape
        channels = [
            '@Shageronlinestore',  # Replace with real channel usernames
            '@example_channel_2',
            '@example_channel_3',
            '@example_channel_4',
            '@example_channel_5'
        ]

        # Iterate over channels and scrape data
        for channel in channels:
            print(f"Scraping channel: {channel}")
            await scrape_channel(client, channel, writer, media_dir)

    print(f"Scraping complete. Data saved to {csv_file} and media in {media_dir}.")

# Run the script
if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
