import os
import smtplib

from time import sleep
from loguru import logger

from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart


class EmailClient:
    """
    Class used to send email's

    """
    host = os.environ.get("EMAIL_HOST", "")
    port = int(os.environ.get("EMAIL_PORT", 123))
    user = os.environ.get("EMAIL_SENDER", "")
    password = os.environ.get("EMAIL_PASSWORD", "")
    recipients = os.environ.get("EMAIL_RECIPIENTS", "").split(',')
    instance = None

    def __init__(self, smtp_ssl=False, omit_recipients=False):
        """
        Connect to mail server

        :param host: (:obj:`str`) Server host.
        :param port: (:obj:`int`) Server port
        :param user: (:obj:`str`) Username required to login in mail server.
        :param password: (:obj:`str`) User password required to login in mail server.
        """
        # -- Connect to Server Mail
        logger.debug(f"Connecting to email server: {self.host} ... ")
        self.omit_recipients = omit_recipients
        self.msg = None
        self.mailserver = None
        self.smtp_ssl = smtp_ssl
        self.connect()
        self.prepare_msg()

    def update_recipient(self, new_recipients_debug):
        """
        Update the recipients_debug list.

        Parameters:
            new_recipients_debug (list): List of new recipients for debugging.
        """
        self.recipients = new_recipients_debug

    @staticmethod
    def get_email_instance():
        if EmailClient.instance is None:
            try:
                EmailClient.instance = EmailClient()
            except BaseException:
                logger.exception("ERROR! Unable to connect to email.")
                logger.debug("Connecting to email server: ... Failed!")
                return None
        return EmailClient.instance

    def connect(self):
        if self.smtp_ssl:
            self.mailserver = self.connect_smtp_ssl(host=self.host, port=self.port,
                                                    user=self.user, password=self.password)
        else:
            self.mailserver = self.connect_smtp(host=self.host, port=self.port,
                                                user=self.user, password=self.password)

    def prepare_msg(self):
        # Create the container email message.
        self.msg = MIMEMultipart()
        if not self.omit_recipients :
            self.msg['To'] = ", ".join(self.recipients)
        logger.debug(f"Connecting to email server: {self.host} ... Ok!")

    @staticmethod
    def connect_smtp_ssl(host, port, user, password):
        import ssl
        # Create a secure SSL context
        context = ssl.create_default_context()
        # Try to log in to server and send email
        server = smtplib.SMTP(host=host, port=port)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(user=user, password=password)
        return server

    @staticmethod
    def connect_smtp(host, port, user, password):
        server = smtplib.SMTP(f'{host}:{port}')
        server.ehlo()
        server.starttls()
        server.login(user=user, password=password)
        return server

    def set_attachments(self, filename, file_path):
        """

        :param filename:
        :param file_path:
        :return:
        """

        attachment = MIMEBase('application', "octet-stream")
        attachment.set_payload(open(file_path, "rb").read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', 'attachment; filename="%s"' % filename)
        self.msg.attach(attachment)

    def set_subject(self, subject):
        self.msg['Subject'] = subject

    def set_body(self, message):
        self.msg.attach(MIMEText(message))

    def send_email(self, retry_on_failure=True, max_retry_attempts=5, retry_wait_time=30):
        """
        Send email

        :param retry_on_failure: (:obj:`bool`) Try to retry to send when exception occurs
        :param max_retry_attempts: (:obj:`int`) Number of retry attempts
        :param retry_wait_time: (:obj:`int`) Number of seconds to wait between each retry

        :return:
        """

        if self.user is None:
            exit("Must define a sender email.")

        attempts = 0
        sending = True

        while sending:
            attempts += 1
            try:
                self.mailserver.sendmail(self.user, self.recipients, self.msg.as_string())
                sending = False
            except BaseException as ex:
                if retry_on_failure and (attempts < max_retry_attempts + 1):
                    print("ERROR! Exception the following exception occurred while sending email:")
                    print(repr(ex))
                    print(f"Retrying in {retry_wait_time} seconds ...")
                    sleep(retry_wait_time)
                else:
                    raise

    def compose_and_send(self, subject, msg, signature=None,file_to_send=None,path_file=None):
        if signature is not None:
            email_body = f"{msg}\n\nBest regards,\n{signature}"
        else:
            email_body = f"{msg}"

        self.set_subject(subject=subject)
        self.set_body(message=email_body)
        if file_to_send is not None:
            self.set_attachments( filename=file_to_send,file_path=path_file)
        self.send_email()

    def ensure_connection(self):
        logger.debug(f'mailserver:{self.mailserver.noop()[0]}')
        if self.mailserver is None or not self.mailserver.noop()[0] == 250:
            self.connect()

    def close(self):
        self.mailserver.quit()
        self.mailserver.close()