#sudo docker run -d --name graphite --restart=always -p 2003-2004:2003-2004 -p 2023-2024:2023-2024 -p 8080:18080 -p 8125:8125/udp -p 8126:8126 graphiteapp/graphite-statsd
sudo docker run -d --restart=always -p 2003-2004:2003-2004 -p 2023-2024:2023-2024 -p 18080:8080 -p 8125:8125/udp -p 8126:8126 graphiteapp/graphite-statsd
