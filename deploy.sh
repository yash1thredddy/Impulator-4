#!/bin/bash

# Simple deployment script for IMPULATOR
# Make this script executable with: chmod +x deploy.sh

# Stop and remove existing containers
echo "Stopping existing containers..."
docker-compose down

# Pull latest changes if using git
if [ -d ".git" ]; then
    echo "Pulling latest changes..."
    git pull
fi

# Build and start the container
echo "Building and starting container..."
docker-compose up -d --build

# Check the container status
echo "Container status:"
docker-compose ps

