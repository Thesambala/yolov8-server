/**
 * ecosystem.config.js
 * PM2 Process Management Configuration for Adaptive Traffic Monitoring
 * 
 * Usage:
 *   pm2 start ecosystem.config.js
 *   pm2 list
 *   pm2 logs
 *   pm2 stop all
 */

module.exports = {
  apps: [
    {
      name: "adaptive-vision-fuzzy-server",
      script: "vision_fuzzy_server.py",
      interpreter: "python",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        PORT: 8081,
        DETECT_MODEL: "/home/ubuntu/models/yolov8s.pt",
        DETECT_CONF: "0.25",
        DETECT_IOU: "0.60",
        DETECT_IMGSZ: "960",
        MIN_GREEN_SEC: "10",
        MAX_GREEN_SEC: "60"
      },
      env_production: {
        PORT: 8081,
        DETECT_MODEL: "/home/ubuntu/models/yolov8s.pt",
        DETECT_CONF: "0.25",
        DETECT_IOU: "0.60",
        DETECT_IMGSZ: "960",
        MIN_GREEN_SEC: "10",
        MAX_GREEN_SEC: "60"
      }
    }
  ]
};
