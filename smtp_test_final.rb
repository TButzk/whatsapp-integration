require "net/smtp"
require "mail"

begin
  smtp_config = {
    address: ENV["SMTP_ADDRESS"],
    port: ENV["SMTP_PORT"].to_i,
    user_name: ENV["SMTP_USERNAME"],
    password: ENV["SMTP_PASSWORD"],
    authentication: (ENV["SMTP_AUTHENTICATION"] || "login").to_sym,
    enable_starttls_auto: true
  }
  
  sender = ENV["SMTP_USERNAME"]
  puts "Testing SMTP with address: #{smtp_config[:address]}, port: #{smtp_config[:port]}, sender: #{sender}"
  
  Mail.defaults do
    delivery_method :smtp, smtp_config
  end

  Mail.deliver do
    to sender
    from sender
    subject "SMTP Test - Chatwoot"
    body "Teste de configuracao SMTP executado com sucesso."
  end
  
  puts "SMTP_ENV_OK"
rescue => e
  puts "SMTP_ERROR: #{e.message}"
  exit 1
end
